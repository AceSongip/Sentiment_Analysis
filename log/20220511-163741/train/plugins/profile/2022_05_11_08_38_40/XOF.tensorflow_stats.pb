"?:
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1????1?AA????1?Aa??I???i??I????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1????u?@9????u?@A????u?@I????u?@a!ϸP????iӘ?2????Unknown?
?HostAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(1?????X?@9?????X?@A?????X?@I?????X?@a???c???i??[|???Unknown
{HostReadVariableOp"Adam/Adam/update/ReadVariableOp(1????̳?@9????̳?@A????̳?@I????̳?@a@???p??i?n???g???Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(1????̞?@9????̞?@A????̞?@I????̞?@a?+??J??i??XRh????Unknown
?HostResourceGather"'sequential_2/embedding/embedding_lookup(133333??@933333??@A33333??@I33333??@aΠD??q?iC??jy????Unknown
HostAssignVariableOp"!Adam/Adam/update/AssignVariableOp(133333??@933333??@A33333??@I33333??@a???8?Lp?i?dS????Unknown
g	HostMul"Adam/Adam/update/mul_4(1?????i?@9?????i?@A?????i?@I?????i?@a???)p?iz??f???Unknown
}
HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1     ??@9     ??@A     ??@I     ??@aQ?Z?ei?i?s$?%???Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1fffffR?@9fffffR?@AfffffR?@IfffffR?@a??????h?i?g	?>???Unknown
gHostMul"Adam/Adam/update/mul_1(1???????@9???????@A???????@I???????@a;?"mƁd?i????+S???Unknown
?HostAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(1fffffz?@9fffffz?@Afffffz?@Ifffffz?@a|??I?*d?i??sVg???Unknown
~Host_Send"+sequential_2/embedding/embedding_lookup/_25(1????̬}@9????̬}@A????̬}@I????̬}@ax?Z<?(b?iX[y???Unknown
dHostDataset"Iterator::Model(1??????~@9??????~@A????̔}@I????̔}@a???b?iH??????Unknown
iHostWriteSummary"WriteSummary(1     hs@9     hs@A     hs@I     hs@a?/???W?i?^by????Unknown?
?HostUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(133333r@933333r@A33333r@I33333r@a?#?Ż(V?ir3Ee?????Unknown
?HostResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1????̄p@9????̄p@A????̄p@I????̄p@a??M?]7T?iKZ?????Unknown
fHost_Send"IteratorGetNext/_13(1??????m@9??????m@A??????m@I??????m@a}????&R?iC-?i?????Unknown
?HostResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(1?????lk@9?????lk@A?????lk@I?????lk@a?4@)?P?i]?~ ????Unknown
eHostMul"Adam/Adam/update/mul(133333?h@933333?h@A33333?h@I33333?h@a???f?:N?i?u?"?????Unknown
kHostUnique"Adam/Adam/update/Unique(133333?e@933333?e@A33333?e@I33333?e@a
\?o??J?i/qbf????Unknown
gHostSqrt"Adam/Adam/update/Sqrt(1?????e@9?????e@A?????e@I?????e@a]??Ϯ?I?i?j??????Unknown
gHostMul"Adam/Adam/update/mul_2(1      `@9      `@A      `@I      `@aKԱהC?iW?C?????Unknown
eHost
LogicalAnd"
LogicalAnd(1?????l_@9?????l_@A?????l_@I?????l_@ao6??:C?i?$???????Unknown?
gHostMul"Adam/Adam/update/mul_5(1?????y]@9?????y]@A?????y]@I?????y]@a?e?K	B?iGhxG????Unknown
{Host_Send"(Adam/Adam/update/AssignSubVariableOp/_36(133333?Y@933333?Y@A33333?Y@I33333?Y@a??33???i??ލ	????Unknown
mHostRealDiv"Adam/Adam/update/truediv(1fffff?S@9fffff?S@Afffff?S@Ifffff?S@aR2?d8?if|????Unknown
gHostAddV2"Adam/Adam/update/add(1?????9S@9?????9S@A?????9S@I?????9S@a|??=?7?i??/?????Unknown
^HostGatherV2"GatherV2(1ffffffO@9ffffffO@AffffffO@IffffffO@a??m?63?i??]?b????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1fffffP@9fffffP@A      G@I      G@a,??&?%,?iz
?<%????Unknown
? Host_Recv"Cgradient_tape/sequential_2/embedding/embedding_lookup/Reshape_1/_28(1     @E@9     @E@A     @E@I     @E@a?-ؠ?*?i??W?????Unknown
?!HostVariableShape"Cgradient_tape/sequential_2/embedding/embedding_lookup/VariableShape(1??????D@9??????D@A??????D@I??????D@a??s[?)?ilRA?]????Unknown
r"Host_Recv"sequential_2/embedding/Cast/_24(1?????D@9?????D@A?????D@I?????D@a;qD??(?i???8?????Unknown
g#HostMul"Adam/Adam/update/mul_3(1     @B@9     @B@A     @B@I     @B@a&??M?U&?i ??K????Unknown
k$Host_Recv"Adam/ReadVariableOp_1/_2(133333?7@933333?7@A33333?7@I33333?7@aBϥ?D?iNr/?3????Unknown
?%Host	_HostSend"Fcategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_17(1?????7@9?????7@A?????7@I?????7@a??۱IE?i.}?????Unknown
`&HostGatherV2"
GatherV2_1(133333?4@933333?4@A33333?4@I33333?4@atw??\U?i?bt?????Unknown
s'HostDataset"Iterator::Model::ParallelMapV2(1?????L3@9?????L3@A?????L3@I?????L3@aH????i;Jj?????Unknown
?(HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?????2@9?????"@A?????2@I?????"@a?^?|?&?i?"~?N????Unknown
p)Host_Recv"Adam/Cast_6/ReadVariableOp/_6(13333331@93333331@A3333331@I3333331@a?E???i????????Unknown
x*HostDataset"#Iterator::Model::ParallelMapV2::Zip(133333?\@933333?\@Afffff?0@Ifffff?0@aա`?Ϯ?i??j}?????Unknown
f+Host_Send"IteratorGetNext/_11(1??????0@9??????0@A??????0@I??????0@aRj{??i??E?@????Unknown
x,HostStridedSlice"Adam/Adam/update/strided_slice(1      0@9      0@A      0@I      0@aKԱה?iV???????Unknown
?-HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1333333>@9333333>@A??????/@I??????/@a?4?-V?i?nQx????Unknown
?.HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????,@9??????,@A??????,@I??????,@a???Z???iK??M????Unknown
[/HostPow"
Adam/Pow_2(1??????%@9??????%@A??????%@I??????%@as?????
?i;r?p????Unknown
l0HostIteratorGetNext"IteratorGetNext(1ffffff"@9ffffff"@Affffff"@Iffffff"@a?M?Ą?i????????Unknown
p1Host_Recv"Adam/Cast_4/ReadVariableOp/_4(1?????? @9?????? @A?????? @I?????? @a??{S?P?i?;5\????Unknown
p2Host_Recv"Adam/Cast_7/ReadVariableOp/_8(1??????@9??????@A??????@I??????@a????i\??c????Unknown
?3Host_Send"Fgradient_tape/sequential_2/embedding/embedding_lookup/VariableShape/_9(1??????@9??????@A??????@I??????@a?Dc?Uo?>i?lb??????Unknown
?4HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1ffffffA@9ffffffA@Affffff@Iffffff@a?M?Ą?>i?????????Unknown
]5HostAddV2"
Adam/add_1(1333333@9333333@A333333@I333333@a?E???>iم??????Unknown
?6Host_Recv"Agradient_tape/sequential_2/embedding/embedding_lookup/Reshape/_34(1      @9      @A      @I      @aKԱה?>ip?3?????Unknown
]7HostCast"Adam/Cast_5(1333333@9333333@A333333@I333333@ac?????>iq?=*=????Unknown
[8HostMul"
Adam/mul_1(1ffffff
@9ffffff
@Affffff
@Iffffff
@a?Yӗ'?>i#dmy]????Unknown
[9HostSub"
Adam/sub_6(1ffffff
@9ffffff
@Affffff
@Iffffff
@a?Yӗ'?>i?
??}????Unknown
c:HostRealDiv"Adam/truediv_1(1333333@9333333@A333333@I333333@a?@(=?d?>i?G;-?????Unknown
[;HostPow"
Adam/Pow_3(1??????@9??????@A??????@I??????@a?Dc?Uo?>i`А??????Unknown
[<HostSub"
Adam/sub_7(1??????@9??????@A??????@I??????@a/? .?t?>ia?A?????Unknown
]=HostSqrt"Adam/Sqrt_1(1????????9????????A????????I????????a???Z???>iNYа?????Unknown
[>HostSub"
Adam/sub_4(1????????9????????A????????I????????a???Z???>i;?^P?????Unknown
[?HostSub"
Adam/sub_5(1????????9????????A????????I????????a?OL?T?>iO??%?????Unknown
a@HostIdentity"Identity(1ffffff??9ffffff??Affffff??Iffffff??a?????i?>i      ???Unknown?*?8
uHostFlushSummaryWriter"FlushSummaryWriter(1????u?@9????u?@A????u?@I????u?@aEc??]H??iEc??]H???Unknown?
?HostAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(1?????X?@9?????X?@A?????X?@I?????X?@a?U^???i?d??????Unknown
{HostReadVariableOp"Adam/Adam/update/ReadVariableOp(1????̳?@9????̳?@A????̳?@I????̳?@a!?^*?P??i_8H],????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(1????̞?@9????̞?@A????̞?@I????̞?@a??e??Ȯ?il?	?????Unknown
?HostResourceGather"'sequential_2/embedding/embedding_lookup(133333??@933333??@A33333??@I33333??@a????:???i????????Unknown
HostAssignVariableOp"!Adam/Adam/update/AssignVariableOp(133333??@933333??@A33333??@I33333??@a?IjO?͞?i<??8????Unknown
gHostMul"Adam/Adam/update/mul_4(1?????i?@9?????i?@A?????i?@I?????i?@a? ?m%???iA????????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1     ??@9     ??@A     ??@I     ??@a-??h????iJ?<??????Unknown
}	HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1fffffR?@9fffffR?@AfffffR?@IfffffR?@a???w???i?&?#?J???Unknown
g
HostMul"Adam/Adam/update/mul_1(1???????@9???????@A???????@I???????@a???`??i?@?????Unknown
?HostAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(1fffffz?@9fffffz?@Afffffz?@Ifffffz?@a???}-??i?Z??}???Unknown
~Host_Send"+sequential_2/embedding/embedding_lookup/_25(1????̬}@9????̬}@A????̬}@I????̬}@a???|(??ie?ܒB???Unknown
dHostDataset"Iterator::Model(1??????~@9??????~@A????̔}@I????̔}@a0Xa???i?Q?u????Unknown
iHostWriteSummary"WriteSummary(1     hs@9     hs@A     hs@I     hs@a????q??i??	??????Unknown?
?HostUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(133333r@933333r@A33333r@I33333r@aڃ*?*???i??>4?=???Unknown
?HostResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1????̄p@9????̄p@A????̄p@I????̄p@amp?M4??iy?u????Unknown
fHost_Send"IteratorGetNext/_13(1??????m@9??????m@A??????m@I??????m@a??31?&??i?p:??????Unknown
?HostResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(1?????lk@9?????lk@A?????lk@I?????lk@a??H????i9X???Unknown
eHostMul"Adam/Adam/update/mul(133333?h@933333?h@A33333?h@I33333?h@aL6ݩ-?|?i??X?-G???Unknown
kHostUnique"Adam/Adam/update/Unique(133333?e@933333?e@A33333?e@I33333?e@aw}%?by?i????y???Unknown
gHostSqrt"Adam/Adam/update/Sqrt(1?????e@9?????e@A?????e@I?????e@an??ofx?i???????Unknown
gHostMul"Adam/Adam/update/mul_2(1      `@9      `@A      `@I      `@a2y[>??r?i?ρ??????Unknown
eHost
LogicalAnd"
LogicalAnd(1?????l_@9?????l_@A?????l_@I?????l_@ak?o??+r?i??????Unknown?
gHostMul"Adam/Adam/update/mul_5(1?????y]@9?????y]@A?????y]@I?????y]@aoA?=?
q?i\r?,???Unknown
{Host_Send"(Adam/Adam/update/AssignSubVariableOp/_36(133333?Y@933333?Y@A33333?Y@I33333?Y@a?[?V:n?i^?/4???Unknown
mHostRealDiv"Adam/Adam/update/truediv(1fffff?S@9fffff?S@Afffff?S@Ifffff?S@a>Wȧ-?f?i??p2?J???Unknown
gHostAddV2"Adam/Adam/update/add(1?????9S@9?????9S@A?????9S@I?????9S@a:鹋\;f?i????#a???Unknown
^HostGatherV2"GatherV2(1ffffffO@9ffffffO@AffffffO@IffffffO@a ??f?'b?i`dcaKs???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1fffffP@9fffffP@A      G@I      G@a8~?9??Z?i6 ֗????Unknown
?Host_Recv"Cgradient_tape/sequential_2/embedding/embedding_lookup/Reshape_1/_28(1     @E@9     @E@A     @E@I     @E@a?|?zגX?iݞ?A?????Unknown
?HostVariableShape"Cgradient_tape/sequential_2/embedding/embedding_lookup/VariableShape(1??????D@9??????D@A??????D@I??????D@a?3mX?iw?@x?????Unknown
r Host_Recv"sequential_2/embedding/Cast/_24(1?????D@9?????D@A?????D@I?????D@a????/W?i??ND?????Unknown
g!HostMul"Adam/Adam/update/mul_3(1     @B@9     @B@A     @B@I     @B@a=V ?U?i?{֠????Unknown
k"Host_Recv"Adam/ReadVariableOp_1/_2(133333?7@933333?7@A33333?7@I33333?7@a?~??#hK?i?????????Unknown
?#Host	_HostSend"Fcategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_17(1?????7@9?????7@A?????7@I?????7@a??փ?J?i֮?J?????Unknown
`$HostGatherV2"
GatherV2_1(133333?4@933333?4@A33333?4@I33333?4@a?WF?G?il@?K?????Unknown
s%HostDataset"Iterator::Model::ParallelMapV2(1?????L3@9?????L3@A?????L3@I?????L3@a?V???QF?i??,????Unknown
?&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?????2@9?????"@A?????2@I?????"@a?zW#Q?D?i?}]Dh????Unknown
p'Host_Recv"Adam/Cast_6/ReadVariableOp/_6(13333331@93333331@A3333331@I3333331@a|U????C?iv??<a????Unknown
x(HostDataset"#Iterator::Model::ParallelMapV2::Zip(133333?\@933333?\@Afffff?0@Ifffff?0@aj???C?i??5D????Unknown
f)Host_Send"IteratorGetNext/_11(1??????0@9??????0@A??????0@I??????0@a?'wmC?i???^????Unknown
x*HostStridedSlice"Adam/Adam/update/strided_slice(1      0@9      0@A      0@I      0@a2y[>??B?iy???????Unknown
?+HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1333333>@9333333>@A??????/@I??????/@a|T?mEB?ix??P????Unknown
?,HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????,@9??????,@A??????,@I??????,@azSRk??@?i?L??z????Unknown
[-HostPow"
Adam/Pow_2(1??????%@9??????%@A??????%@I??????%@a??\۩59?i=?&V?????Unknown
l.HostIteratorGetNext"IteratorGetNext(1ffffff"@9ffffff"@Affffff"@Iffffff"@a?1?? G5?icF:J????Unknown
p/Host_Recv"Adam/Cast_4/ReadVariableOp/_4(1?????? @9?????? @A?????? @I?????? @aX?~?A23?i@Ń??????Unknown
p0Host_Recv"Adam/Cast_7/ReadVariableOp/_8(1??????@9??????@A??????@I??????@a???b1?i???.?????Unknown
?1Host_Send"Fgradient_tape/sequential_2/embedding/embedding_lookup/VariableShape/_9(1??????@9??????@A??????@I??????@a8}??t?(?i??)?c????Unknown
?2HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1ffffffA@9ffffffA@Affffff@Iffffff@a?1?? G%?i?9H?????Unknown
]3HostAddV2"
Adam/add_1(1333333@9333333@A333333@I333333@a|U????#?iDPS??????Unknown
?4Host_Recv"Agradient_tape/sequential_2/embedding/embedding_lookup/Reshape/_34(1      @9      @A      @I      @a2y[>??"?i?5w?????Unknown
]5HostCast"Adam/Cast_5(1333333@9333333@A333333@I333333@a?/??7
"?i???3?????Unknown
[6HostMul"
Adam/mul_1(1ffffff
@9ffffff
@Affffff
@Iffffff
@a_?r??iF??o3????Unknown
[7HostSub"
Adam/sub_6(1ffffff
@9ffffff
@Affffff
@Iffffff
@a_?r??i?p?'????Unknown
c8HostRealDiv"Adam/truediv_1(1333333@9333333@A333333@I333333@a??t??i"	L?????Unknown
[9HostPow"
Adam/Pow_3(1??????@9??????@A??????@I??????@a8}??t??i???????Unknown
[:HostSub"
Adam/sub_7(1??????@9??????@A??????@I??????@a[?v???i?ԫ??????Unknown
];HostSqrt"Adam/Sqrt_1(1????????9????????A????????I????????azSRk???iH/o?????Unknown
[<HostSub"
Adam/sub_4(1????????9????????A????????I????????azSRk???iۉ2??????Unknown
[=HostSub"
Adam/sub_5(1????????9????????A????????I????????a?[?0???>i ?l1?????Unknown
a>HostIdentity"Identity(1ffffff??9ffffff??Affffff??Iffffff??a??I??>i      ???Unknown?